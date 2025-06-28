# -*- coding: utf-8 -*-
"""Daily scheduler for running forecasts and strategies."""

import logging
import sys
import json
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from functools import wraps

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Project imports
# from core.agents.router import AgentRouter  # Moved to archive
from trading.llm.llm_interface import LLMInterface
from core.agents.self_improving_agent import SelfImprovingAgent
from trading.memory.performance_logger import log_performance
from trading.utils.logging import setup_logging
from core.agents.goal_planner import evaluate_goals
from trading.config.settings import (
    SCHEDULER_LOG_PATH,
    TICKER_CONFIG_PATH,
    TICKER_API_ENDPOINT,
    TICKER_SOURCE,
    MAX_CONCURRENT_TICKERS
)

# Configure logging
logger = setup_logging("daily_scheduler", SCHEDULER_LOG_PATH)

@dataclass
class TickerResult:
    """Result of ticker analysis."""
    ticker: str
    success: bool
    forecast_metrics: Optional[Dict[str, Any]] = None
    strategy_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary with fallback.
    
    Args:
        data: Dictionary to get value from
        key: Key to look up
        default: Default value if key not found
        
    Returns:
        Value from dictionary or default
    """
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default

async def load_tickers() -> List[str]:
    """Load tickers from configured source.
    
    Returns:
        List of ticker symbols
    """
    try:
        if TICKER_SOURCE == "file":
            # Load from JSON file
            with open(TICKER_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                return config.get("tickers", [])
                
        elif TICKER_SOURCE == "api":
            # Load from API
            async with aiohttp.ClientSession() as session:
                async with session.get(TICKER_API_ENDPOINT) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("tickers", [])
                    else:
                        logger.error(f"API returned status {response.status}")
                        return []
                        
        elif TICKER_SOURCE == "monitor":
            # Watch for changes in ticker file
            ticker_file = Path(TICKER_CONFIG_PATH)
            if ticker_file.exists():
                with open(ticker_file, 'r') as f:
                    config = json.load(f)
                    return config.get("tickers", [])
            return []
            
        else:
            logger.error(f"Unknown ticker source: {TICKER_SOURCE}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        return []

async def run_ticker_analysis(ticker: str, router: AgentRouter) -> TickerResult:
    """Run forecast and strategy analysis for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        router: Agent router instance
        
    Returns:
        TickerResult containing analysis results
    """
    try:
        logger.info(f"Starting analysis for {ticker}")
        
        # Run forecast
        forecast_result = await router.handle_prompt(
            f"Forecast {ticker} for the next 30 days",
            data=None  # Data will be fetched by the router
        )
        
        if forecast_result.status == "error":
            return TickerResult(
                ticker=ticker,
                success=False,
                error=f"Forecast failed: {forecast_result.error}"
            )
            
        # Run strategy
        strategy_result = await router.handle_prompt(
            f"Apply active strategy to {ticker}",
            data=None  # Data will be fetched by the router
        )
        
        if strategy_result.status == "error":
            return TickerResult(
                ticker=ticker,
                success=False,
                error=f"Strategy failed: {strategy_result.error}"
            )
            
        # Extract metrics safely
        forecast_metrics = {
            "model": safe_get(forecast_result.metadata, "model", "unknown"),
            "mse": safe_get(forecast_result.data, "mse"),
            "accuracy": safe_get(forecast_result.data, "accuracy")
        }
        
        strategy_metrics = {
            "strategy": safe_get(strategy_result.metadata, "strategy", "unknown"),
            "sharpe": safe_get(strategy_result.data, "sharpe"),
            "drawdown": safe_get(strategy_result.data, "drawdown")
        }
        
        # Log performance metrics
        log_performance(
            ticker=ticker,
            model=forecast_metrics["model"],
            strategy=strategy_metrics["strategy"],
            sharpe=strategy_metrics["sharpe"],
            drawdown=strategy_metrics["drawdown"],
            mse=forecast_metrics["mse"],
            accuracy=forecast_metrics["accuracy"],
            notes=f"Daily run at {datetime.now().isoformat()}"
        )
        
        logger.info(f"Successfully completed analysis for {ticker}")
        
        return TickerResult(
            ticker=ticker,
            success=True,
            forecast_metrics=forecast_metrics,
            strategy_metrics=strategy_metrics
        )
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        return TickerResult(
            ticker=ticker,
            success=False,
            error=str(e)
        )

async def run_daily_schedule(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the daily schedule for all tickers.
    
    Args:
        tickers: Optional list of tickers to analyze
        
    Returns:
        Dictionary containing schedule results
    """
    try:
        logger.info("Starting daily schedule")
        
        # Initialize components
        llm = LLMInterface()
        router = AgentRouter(llm)
        
        # Get tickers to analyze
        if not tickers:
            tickers = await load_tickers()
            
        if not tickers:
            logger.warning("No tickers to analyze")
            return {"success": False, "error": "No tickers available"}
            
        logger.info(f"Analyzing tickers: {', '.join(tickers)}")
        
        # Run analysis for all tickers concurrently
        tasks = [
            run_ticker_analysis(ticker, router)
            for ticker in tickers
        ]
        
        # Process in batches to limit concurrency
        results = []
        for i in range(0, len(tasks), MAX_CONCURRENT_TICKERS):
            batch = tasks[i:i + MAX_CONCURRENT_TICKERS]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
        # Calculate statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Run self-improvement
        logger.info("Running self-improvement analysis")
        improvement_agent = SelfImprovingAgent()
        await improvement_agent.analyze_performance()

        # Evaluate goals
        logger.info("Evaluating goal planner status")
        evaluate_goals()
        
        # Log summary
        summary = {
            "total_tickers": len(tickers),
            "successful": len(successful),
            "failed": len(failed),
            "failed_tickers": [r.ticker for r in failed],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Daily schedule completed: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Error in daily schedule: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main entry point for the script."""
    try:
        results = asyncio.run(run_daily_schedule())
        
        # Exit with error code if any tickers failed
        if not results["success"] or results.get("failed", 0) > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Schedule interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
