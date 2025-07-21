#!/usr/bin/env python3
"""
Launch Script for Agent API Service

Starts the Agent API Service with WebSocket support for real-time agent updates.
Enhanced with endpoint logging and execution time tracking.
"""

import asyncio
import functools
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from trading.services.agent_api_service import AgentAPIService

# Add the trading directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class APILoggingMiddleware:
    """Middleware for logging API endpoint calls and execution times."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.endpoint_stats = {}
        self.request_count = 0

    def log_endpoint_call(self, endpoint: str, method: str = "GET"):
        """Decorator to log endpoint calls and execution time."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._log_execution(
                    func, endpoint, method, *args, **kwargs
                )

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._log_execution_sync(func, endpoint, method, *args, **kwargs)

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    async def _log_execution(
        self, func: Callable, endpoint: str, method: str, *args, **kwargs
    ):
        """Log execution of async function."""
        start_time = time.time()
        request_id = self._generate_request_id()

        # Log request start
        self.logger.info(
            f"🚀 API Request Started | ID: {request_id} | {method} {endpoint}"
        )

        try:
            # Extract request details if available
            request_details = self._extract_request_details(args, kwargs)
            if request_details:
                self.logger.debug(
                    f"📋 Request Details | ID: {request_id} | {json.dumps(request_details, default=str)}"
                )

            # Execute function
            result = await func(*args, **kwargs)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log successful completion
            self.logger.info(
                f"✅ API Request Completed | ID: {request_id} | {method} {endpoint} | Time: {execution_time:.3f}s"
            )

            # Update statistics
            self._update_stats(endpoint, method, execution_time, True)

            return result

        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time

            # Log error
            self.logger.error(
                f"❌ API Request Failed | ID: {request_id} | {method} {endpoint} | Time: {
                    execution_time:.3f}s | Error: {
                    str(e)}")

            # Update statistics
            self._update_stats(endpoint, method, execution_time, False)

            raise

    def _log_execution_sync(
        self, func: Callable, endpoint: str, method: str, *args, **kwargs
    ):
        """Log execution of sync function."""
        start_time = time.time()
        request_id = self._generate_request_id()

        # Log request start
        self.logger.info(
            f"🚀 API Request Started | ID: {request_id} | {method} {endpoint}"
        )

        try:
            # Extract request details if available
            request_details = self._extract_request_details(args, kwargs)
            if request_details:
                self.logger.debug(
                    f"📋 Request Details | ID: {request_id} | {json.dumps(request_details, default=str)}"
                )

            # Execute function
            result = func(*args, **kwargs)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log successful completion
            self.logger.info(
                f"✅ API Request Completed | ID: {request_id} | {method} {endpoint} | Time: {execution_time:.3f}s"
            )

            # Update statistics
            self._update_stats(endpoint, method, execution_time, True)

            return result

        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time

            # Log error
            self.logger.error(
                f"❌ API Request Failed | ID: {request_id} | {method} {endpoint} | Time: {
                    execution_time:.3f}s | Error: {
                    str(e)}")

            # Update statistics
            self._update_stats(endpoint, method, execution_time, False)

            raise

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self.request_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"req_{timestamp}_{self.request_count:06d}"

    def _extract_request_details(self, args, kwargs) -> Optional[Dict[str, Any]]:
        """Extract relevant request details for logging."""
        details = {}

        # Look for common request objects
        for arg in args:
            if hasattr(arg, "method") and hasattr(arg, "url"):
                details["method"] = arg.method
                details["url"] = str(arg.url)
                if hasattr(arg, "headers"):
                    details["headers"] = dict(arg.headers)
                if hasattr(arg, "query_params"):
                    details["query_params"] = dict(arg.query_params)
                break

        # Extract from kwargs
        for key, value in kwargs.items():
            if key in ["data", "params", "headers"] and value:
                details[key] = value

        return details if details else None

    def _update_stats(
        self, endpoint: str, method: str, execution_time: float, success: bool
    ):
        """Update endpoint statistics."""
        key = f"{method} {endpoint}"

        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "last_called": None,
            }

        stats = self.endpoint_stats[key]
        stats["total_calls"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["total_calls"]
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["last_called"] = datetime.now().isoformat()

        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of endpoint statistics."""
        summary = {"total_requests": self.request_count, "endpoints": {}}

        for endpoint, stats in self.endpoint_stats.items():
            summary["endpoints"][endpoint] = {
                "total_calls": stats["total_calls"],
                "success_rate": (
                    stats["successful_calls"] / stats["total_calls"]
                    if stats["total_calls"] > 0
                    else 0
                ),
                "avg_response_time": stats["avg_time"],
                "min_response_time": (
                    stats["min_time"] if stats["min_time"] != float("inf") else 0
                ),
                "max_response_time": stats["max_time"],
                "last_called": stats["last_called"],
            }

        return summary

    def log_stats_summary(self):
        """Log endpoint statistics summary."""
        summary = self.get_stats_summary()

        self.logger.info("📊 API Endpoint Statistics Summary")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Requests: {summary['total_requests']}")

        for endpoint, stats in summary["endpoints"].items():
            self.logger.info(f"\n{endpoint}:")
            self.logger.info(f"  Total Calls: {stats['total_calls']}")
            self.logger.info(f"  Success Rate: {stats['success_rate']:.1%}")
            self.logger.info(f"  Avg Response Time: {stats['avg_response_time']:.3f}s")
            self.logger.info(
                f"  Response Time Range: {stats['min_response_time']:.3f}s - {stats['max_response_time']:.3f}s"
            )
            self.logger.info(f"  Last Called: {stats['last_called']}")


def setup_logging():
    """Set up logging for the launch script."""
    log_path = Path("logs/api")
    log_path.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler with rotation
    file_handler = logging.FileHandler(log_path / "agent_api_launch.log")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Create API-specific logger
    api_logger = logging.getLogger("agent_api")
    api_logger.setLevel(logging.INFO)

    return api_logger


async def main():
    """Main entry point."""
    logger = setup_logging()

    # Initialize logging middleware
    api_middleware = APILoggingMiddleware(logger)

    try:
        logger.info("🚀 Starting Agent API Service...")

        # Initialize and start the service with middleware
        service = AgentAPIService()

        # Apply middleware to service endpoints
        service.apply_middleware(api_middleware)

        logger.info("✅ Agent API Service initialized successfully")
        logger.info("🌐 Available endpoints:")
        logger.info("  - REST API: http://localhost:8001")
        logger.info("  - API Docs: http://localhost:8001/docs")
        logger.info("  - WebSocket: ws://localhost:8001/ws")
        logger.info("  - WebSocket Test: http://localhost:8001/ws/test")
        logger.info("  - Health Check: http://localhost:8001/health")
        logger.info("  - Stats: http://localhost:8001/stats")

        # Start the service
        await service.start()

        # Log periodic statistics
        while True:
            await asyncio.sleep(300)  # Log stats every 5 minutes
            api_middleware.log_stats_summary()

    except KeyboardInterrupt:
        logger.info("🛑 Agent API Service interrupted by user")
        # Log final statistics
        api_middleware.log_stats_summary()
    except Exception as e:
        logger.error(f"💥 Error starting Agent API Service: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
