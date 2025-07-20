"""
Prompt Router Example

This example demonstrates how the updated prompt router uses TaskAgent
for executing tasks instead of direct agent calls.
"""

import asyncio
import logging
from typing import Dict, Any

from routing.prompt_router import get_prompt_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_forecast_prompt():
    """Example of processing a forecast prompt."""
    logger.info("=== Example: Forecast Prompt ===")
    
    prompt = "Build a forecasting model for AAPL stock price prediction with LSTM"
    
    router = get_prompt_router()
    result = await router.route_prompt(prompt, user_id="test_user")
    
    logger.info(f"Success: {result['success']}")
    logger.info(f"Routing Type: {result['routing_type']}")
    logger.info(f"Task Type: {result.get('task_type', 'N/A')}")
    logger.info(f"Performance Score: {result.get('performance_score', 'N/A')}")
    logger.info(f"Message: {result.get('message', 'N/A')}")
    
    if result.get('task_id'):
        logger.info(f"Task ID: {result['task_id']}")
    
    return result


async def example_strategy_prompt():
    """Example of processing a strategy prompt."""
    logger.info("\n=== Example: Strategy Prompt ===")
    
    prompt = "Create a momentum-based trading strategy for SPY ETF using ensemble models"
    
    router = get_prompt_router()
    result = await router.route_prompt(prompt, user_id="test_user")
    
    logger.info(f"Success: {result['success']}")
    logger.info(f"Routing Type: {result['routing_type']}")
    logger.info(f"Task Type: {result.get('task_type', 'N/A')}")
    logger.info(f"Performance Score: {result.get('performance_score', 'N/A')}")
    logger.info(f"Message: {result.get('message', 'N/A')}")
    
    if result.get('task_id'):
        logger.info(f"Task ID: {result['task_id']}")
    
    return result


async def example_backtest_prompt():
    """Example of processing a backtest prompt."""
    logger.info("\n=== Example: Backtest Prompt ===")
    
    prompt = "Run a comprehensive backtest of a mean reversion strategy on QQQ data"
    
    router = get_prompt_router()
    result = await router.route_prompt(prompt, user_id="test_user")
    
    logger.info(f"Success: {result['success']}")
    logger.info(f"Routing Type: {result['routing_type']}")
    logger.info(f"Task Type: {result.get('task_type', 'N/A')}")
    logger.info(f"Performance Score: {result.get('performance_score', 'N/A')}")
    logger.info(f"Message: {result.get('message', 'N/A')}")
    
    if result.get('task_id'):
        logger.info(f"Task ID: {result['task_id']}")
    
    return result


async def example_builder_workflow():
    """Example of processing a builder workflow prompt."""
    logger.info("\n=== Example: Builder Workflow ===")
    
    prompt = "Build a new LSTM model for cryptocurrency price prediction"
    
    router = get_prompt_router()
    result = await router.route_prompt(prompt, user_id="test_user")
    
    logger.info(f"Success: {result['success']}")
    logger.info(f"Routing Type: {result['routing_type']}")
    logger.info(f"Workflow Type: {result.get('workflow_type', 'N/A')}")
    logger.info(f"Task Type: {result.get('task_type', 'N/A')}")
    logger.info(f"Performance Score: {result.get('performance_score', 'N/A')}")
    logger.info(f"Message: {result.get('message', 'N/A')}")
    
    return result


async def example_evaluator_workflow():
    """Example of processing an evaluator workflow prompt."""
    logger.info("\n=== Example: Evaluator Workflow ===")
    
    prompt = "Evaluate the performance of the latest model and provide detailed metrics"
    
    router = get_prompt_router()
    result = await router.route_prompt(prompt, user_id="test_user")
    
    logger.info(f"Success: {result['success']}")
    logger.info(f"Routing Type: {result['routing_type']}")
    logger.info(f"Workflow Type: {result.get('workflow_type', 'N/A')}")
    logger.info(f"Task Type: {result.get('task_type', 'N/A')}")
    logger.info(f"Performance Score: {result.get('performance_score', 'N/A')}")
    logger.info(f"Message: {result.get('message', 'N/A')}")
    
    return result


async def example_updater_workflow():
    """Example of processing an updater workflow prompt."""
    logger.info("\n=== Example: Updater Workflow ===")
    
    prompt = "Update and optimize the current model based on recent performance data"
    
    router = get_prompt_router()
    result = await router.route_prompt(prompt, user_id="test_user")
    
    logger.info(f"Success: {result['success']}")
    logger.info(f"Routing Type: {result['routing_type']}")
    logger.info(f"Workflow Type: {result.get('workflow_type', 'N/A')}")
    logger.info(f"Task Type: {result.get('task_type', 'N/A')}")
    logger.info(f"Performance Score: {result.get('performance_score', 'N/A')}")
    logger.info(f"Message: {result.get('message', 'N/A')}")
    
    return result


async def example_full_pipeline():
    """Example of processing a full pipeline prompt."""
    logger.info("\n=== Example: Full Pipeline ===")
    
    prompt = "Run the complete pipeline: build, evaluate, and update a model for TSLA stock"
    
    router = get_prompt_router()
    result = await router.route_prompt(prompt, user_id="test_user")
    
    logger.info(f"Success: {result['success']}")
    logger.info(f"Routing Type: {result['routing_type']}")
    logger.info(f"Workflow Type: {result.get('workflow_type', 'N/A')}")
    logger.info(f"Task Type: {result.get('task_type', 'N/A')}")
    logger.info(f"Performance Score: {result.get('performance_score', 'N/A')}")
    logger.info(f"Message: {result.get('message', 'N/A')}")
    
    return result


async def main():
    """Run all examples."""
    logger.info("Starting Prompt Router Examples with TaskAgent Integration")
    logger.info("=" * 60)
    
    try:
        # Test complex task prompts
        forecast_result = await example_forecast_prompt()
        strategy_result = await example_strategy_prompt()
        backtest_result = await example_backtest_prompt()
        
        # Test workflow prompts
        builder_result = await example_builder_workflow()
        evaluator_result = await example_evaluator_workflow()
        updater_result = await example_updater_workflow()
        pipeline_result = await example_full_pipeline()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Forecast Prompt: {'âœ“' if forecast_result['success'] else 'âœ—'}")
        logger.info(f"Strategy Prompt: {'âœ“' if strategy_result['success'] else 'âœ—'}")
        logger.info(f"Backtest Prompt: {'âœ“' if backtest_result['success'] else 'âœ—'}")
        logger.info(f"Builder Workflow: {'âœ“' if builder_result['success'] else 'âœ—'}")
        logger.info(f"Evaluator Workflow: {'âœ“' if evaluator_result['success'] else 'âœ—'}")
        logger.info(f"Updater Workflow: {'âœ“' if updater_result['success'] else 'âœ—'}")
        logger.info(f"Full Pipeline: {'âœ“' if pipeline_result['success'] else 'âœ—'}")
        
        # Show routing statistics
        router = get_prompt_router()
        if hasattr(router, 'task_agent') and router.task_agent:
            all_tasks = router.task_agent.get_all_tasks()
            logger.info(f"\nTotal tasks executed: {len(all_tasks)}")
            
            # Show task types
            task_types = {}
            for task in all_tasks:
                task_type = task.get('task_type', 'unknown')
                task_types[task_type] = task_types.get(task_type, 0) + 1
            
            logger.info("Task type distribution:")
            for task_type, count in task_types.items():
                logger.info(f"  {task_type}: {count}")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
