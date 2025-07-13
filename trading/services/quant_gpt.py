"""
QuantGPT Interface

A natural language interface for the Evolve trading system that provides
GPT-powered commentary on trading decisions and model recommendations.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from trading.memory.agent_memory import AgentMemory
from trading.services.action_executor import ActionExecutor
from trading.services.cache_manager import CacheManager
from trading.services.commentary_generator import CommentaryGenerator
from trading.services.config import QuantGPTConfig
from trading.services.exceptions import (
    ActionExecutionError,
    CommentaryGenerationError,
    ConfigurationError,
    QuantGPTException,
    QueryParsingError,
    RateLimitExceededError,
    ValidationError,
)
from trading.services.query_parser import QueryParser
from trading.services.rate_limiter import RateLimiter
from trading.services.service_client import ServiceClient

logger = logging.getLogger(__name__)


class QuantGPT:
    """
    Natural language interface for the Evolve trading system.

    Provides GPT-powered analysis and commentary on trading decisions,
    model recommendations, and market analysis with advanced features
    including caching, rate limiting, and comprehensive error handling.
    """

    def __init__(
        self,
        config: Optional[QuantGPTConfig] = None,
        openai_api_key: str = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        """
        Initialize QuantGPT with advanced configuration.

        Args:
            config: Configuration object (if None, will load from env/file)
            openai_api_key: OpenAI API key for GPT commentary
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
        """
        # Load configuration
        if config is None:
            config = QuantGPTConfig.from_env()

        # Override config with direct parameters if provided
        if openai_api_key:
            config.openai_api_key = openai_api_key
        if redis_host != "localhost":
            config.redis_host = redis_host
        if redis_port != 6379:
            config.redis_port = redis_port
        if redis_db != 0:
            config.redis_db = redis_db

        # Validate configuration
        if not config.validate():
            raise ConfigurationError("Invalid configuration")

        self.config = config

        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level), format=config.log_format)

        # Initialize service client
        self.client = ServiceClient(config.redis_host, config.redis_port, config.redis_db)

        # Initialize memory system
        self.memory = AgentMemory()

        # Initialize cache manager
        self.cache_manager = CacheManager(
            redis_client=self.client.redis_client if hasattr(self.client, "redis_client") else None,
            cache_enabled=config.cache_enabled,
            ttl=config.cache_ttl,
        )

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(max_calls=config.rate_limit_calls, time_window=config.rate_limit_period)

        # Initialize modular components
        self.query_parser = QueryParser(config.openai_api_key)
        self.action_executor = ActionExecutor(self.client)
        self.commentary_generator = CommentaryGenerator(config.openai_api_key)

        logger.info("QuantGPT initialized with advanced features")
        logger.info(f"Configuration: {self.config.to_dict()}")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query and return comprehensive analysis.

        Args:
            query: Natural language query (e.g., "Give me the best model for NVDA over 90 days")

        Returns:
            Dictionary containing analysis results and GPT commentary
        """
        # Validate query
        if self.config.validate_inputs:
            if not query or not isinstance(query, str):
                raise ValidationError("Invalid query: must be a non-empty string", "query", query)

            query = query.strip()
            if not query:
                raise ValidationError("Query cannot be empty", "query", query)

        # Check rate limit
        if self.config.rate_limit_enabled:
            if not self.rate_limiter.record_call("query_processing"):
                raise RateLimitExceededError("Query processing rate limit exceeded")

        try:
            logger.info(f"Processing query: {query}")

            # Try to get from cache first
            cache_key = f"query_result:{hash(query)}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result

            # Parse the query to extract intent and parameters
            parsed = self._parse_query_with_retry(query)

            # Execute the appropriate action based on intent
            result = self._execute_action_with_retry(parsed)

            # Generate GPT commentary
            commentary = self._generate_commentary_with_retry(query, parsed, result)

            # Log the interaction
            self._log_decision(query, parsed)

            # Prepare final result
            final_result = {
                "query": query,
                "parsed_intent": parsed,
                "results": result,
                "gpt_commentary": commentary,
                "timestamp": time.time(),
                "status": "success",
                "cache_hit": False,
            }

            # Cache the result
            self.cache_manager.set(cache_key, final_result)

            return final_result

        except QuantGPTException:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing query: {e}")
            raise ActionExecutionError(f"Unexpected error: {str(e)}")

    def _parse_query_with_retry(self, query: str) -> Dict[str, Any]:
        """Parse query with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                return self.query_parser.parse_query(query)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise QueryParsingError(
                        f"Failed to parse query after {self.config.max_retries} attempts: {e}", query
                    )
                time.sleep(self.config.retry_delay)

    def _execute_action_with_retry(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                return self.action_executor.execute_action(parsed)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise ActionExecutionError(
                        f"Failed to execute action after {self.config.max_retries} attempts: {e}", parsed.get("intent")
                    )
                time.sleep(self.config.retry_delay)

    def _generate_commentary_with_retry(self, query: str, parsed: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Generate commentary with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                return self.commentary_generator.generate_commentary(query, parsed, result)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise CommentaryGenerationError(
                        f"Failed to generate commentary after {self.config.max_retries} attempts: {e}"
                    )
                time.sleep(self.config.retry_delay)

    def _log_decision(self, query: str, parsed: Dict[str, Any]) -> None:
        """Log decision with error handling."""
        try:
            self.memory.log_decision(
                agent_name="quant_gpt",
                decision_type="query_processed",
                details={
                    "query": query,
                    "intent": parsed.get("intent"),
                    "symbol": parsed.get("symbol"),
                    "timeframe": parsed.get("timeframe"),
                    "period": parsed.get("period"),
                    "confidence": parsed.get("confidence", 0),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with system statistics
        """
        return {
            "cache_stats": self.cache_manager.get_stats(),
            "rate_limit_status": {"query_processing": self.rate_limiter.get_status("query_processing")},
            "configuration": self.config.to_dict(),
        }

    def clear_cache(self) -> bool:
        """
        Clear all cached data.

        Returns:
            True if successful
        """
        return self.cache_manager.clear()

    def reset_rate_limits(self) -> None:
        """Reset all rate limits."""
        self.rate_limiter.reset_all()

    def close(self):
        """Close the QuantGPT interface and clean up resources."""
        try:
            self.client.close()
            logger.info("QuantGPT resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Alias for backward compatibility
QuantGPTAgent = QuantGPT


def main() -> Dict[str, Any]:
    """Main function for command-line usage."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="QuantGPT - Natural Language Trading Interface")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--config-file", help="Configuration file path")
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")

    args = parser.parse_args()

    try:
        # Load configuration
        if args.config_file:
            config = QuantGPTConfig.from_file(args.config_file)
        else:
            config = QuantGPTConfig.from_env()

        # Initialize QuantGPT
        quant_gpt = QuantGPT(
            config=config, openai_api_key=args.openai_key, redis_host=args.redis_host, redis_port=args.redis_port
        )

        if args.stats:
            # Show statistics
            stats = quant_gpt.get_stats()
            print(json.dumps(stats, indent=2))
            return {"status": "completed", "result": stats}

        # Process the query
        result = quant_gpt.process_query(args.query)

        # Print results
        print(json.dumps(result, indent=2))

        return {"status": "completed", "query": args.query, "result": result}

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return {"status": "interrupted", "query": args.query, "result": "user_interrupted"}
    except QuantGPTException as e:
        logger.error(f"QuantGPT error: {e.message}")
        return {
            "status": "failed",
            "query": args.query,
            "error": e.message,
            "error_code": e.error_code,
            "recoverable": e.recoverable,
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"status": "failed", "query": args.query, "error": str(e)}
    finally:
        if "quant_gpt" in locals():
            quant_gpt.close()


if __name__ == "__main__":
    main()
