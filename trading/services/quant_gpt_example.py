#!/usr/bin/env python3
"""
QuantGPT Usage Example

Demonstrates how to use the QuantGPT interface for natural language
trading queries.
"""

import logging
import os
import sys
import time
from pathlib import Path

from services.quant_gpt import QuantGPT
from services.service_client import ServiceClient

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


def example_queries():
    """Run example queries through QuantGPT."""

    logger.info("🤖 QuantGPT Trading Interface Examples")
    logger.info("=" * 60)

    # Initialize QuantGPT (direct usage)
    logger.info("Initializing QuantGPT...")
    quant_gpt = QuantGPT(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        redis_host="localhost",
        redis_port=6379,
    )

    # Example queries
    queries = [
        "Give me the best model for NVDA over 90 days",
        "Should I long TSLA this week?",
        "Analyze BTCUSDT market conditions",
        "What's the trading signal for AAPL?",
        "Find the optimal model for GOOGL on 1h timeframe",
    ]

    logger.info(f"\nRunning {len(queries)} example queries...")
    logger.info("-" * 60)

    for i, query in enumerate(queries, 1):
        logger.info(f"\n📝 Query {i}: {query}")
        logger.info("-" * 40)

        try:
            # Process the query
            start_time = time.time()
            result = quant_gpt.process_query(query)
            processing_time = time.time() - start_time

            # Display results
            if result.get("status") == "success":
                parsed = result.get("parsed_intent", {})
                results = result.get("results", {})
                commentary = result.get("gpt_commentary", "")

                logger.info(f"✅ Success (processed in {processing_time:.2f}s)")
                logger.info(f"Intent: {parsed.get('intent', 'unknown')}")
                logger.info(f"Symbol: {parsed.get('symbol', 'N/A')}")
                logger.info(f"Timeframe: {parsed.get('timeframe', 'N/A')}")
                logger.info(f"Period: {parsed.get('period', 'N/A')}")

                # Display action-specific results
                action = results.get("action", "unknown")
                logger.info(f"Action: {action}")

                if action == "model_recommendation":
                    best_model = results.get("best_model")
                    if best_model:
                        logger.info(f"Best Model: {best_model['model_type'].upper()}")
                        logger.info(
                            f"Model Score: {best_model['evaluation'].get('overall_score', 0):.2f}"
                        )

                elif action == "trading_signal":
                    signal = results.get("signal", {})
                    if signal:
                        logger.info(f"Signal: {signal['signal']}")
                        logger.info(f"Strength: {signal['strength']}")
                        logger.info(f"Confidence: {signal['confidence']:.1%}")

                # Display GPT commentary
                if commentary:
                    logger.info(f"\n🤖 GPT Commentary:")
                    logger.info("-" * 30)
                    logger.info(commentary)

            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Error: {error}")

        except Exception as e:
            logger.error(f"❌ Exception: {e}")

        logger.info("\n" + "=" * 60)

    # Clean up
    quant_gpt.close()


def example_service_client():
    """Demonstrate QuantGPT usage through ServiceClient."""

    logger.info("\n🔗 QuantGPT via ServiceClient Example")
    logger.info("=" * 60)

    # Initialize ServiceClient
    logger.info("Initializing ServiceClient...")
    client = ServiceClient(redis_host="localhost", redis_port=6379)

    # Example queries for service client
    service_queries = [
        "What's the best model for MSFT?",
        "Should I buy AMZN now?",
        "Analyze ETHUSDT market",
    ]

    logger.info(f"\nRunning {len(service_queries)} queries via ServiceClient...")
    logger.info("-" * 60)

    for i, query in enumerate(service_queries, 1):
        logger.info(f"\n📝 Service Query {i}: {query}")
        logger.info("-" * 40)

        try:
            # Process query via service
            start_time = time.time()
            result = client.process_natural_language_query(query)
            processing_time = time.time() - start_time

            if result:
                logger.info(
                    f"✅ Service Response (processed in {processing_time:.2f}s)"
                )
                logger.info(f"Type: {result.get('type', 'unknown')}")

                if result.get("type") == "query_processed":
                    query_result = result.get("result", {})
                    if query_result.get("status") == "success":
                        parsed = query_result.get("parsed_intent", {})
                        logger.info(f"Intent: {parsed.get('intent', 'unknown')}")
                        logger.info(f"Symbol: {parsed.get('symbol', 'N/A')}")

                        commentary = query_result.get("gpt_commentary", "")
                        if commentary:
                            logger.info(f"\n🤖 GPT Commentary:")
                            logger.info("-" * 30)
                            logger.info(
                                commentary[:200] + "..."
                                if len(commentary) > 200
                                else commentary
                            )
                    else:
                        logger.error(
                            f"❌ Query Error: {query_result.get('error', 'Unknown error')}"
                        )
                else:
                    logger.error(
                        f"❌ Service Error: {result.get('error', 'Unknown error')}"
                    )
            else:
                logger.error("❌ No response from service")

        except Exception as e:
            logger.error(f"❌ Exception: {e}")

        logger.info("\n" + "=" * 60)

    # Get available symbols
    logger.info("\n📊 Available Symbols and Parameters")
    logger.info("-" * 40)

    try:
        symbols_result = client.get_available_symbols()
        if symbols_result and symbols_result.get("type") == "available_symbols":
            symbols = symbols_result.get("symbols", [])
            timeframes = symbols_result.get("timeframes", [])
            periods = symbols_result.get("periods", [])
            models = symbols_result.get("models", [])

            logger.info(f"Symbols: {', '.join(symbols)}")
            logger.info(f"Timeframes: {', '.join(timeframes)}")
            logger.info(f"Periods: {', '.join(periods)}")
            logger.info(f"Models: {', '.join(models)}")
        else:
            logger.error("❌ Could not retrieve available symbols")

    except Exception as e:
        logger.error(f"❌ Exception: {e}")

    # Clean up
    client.close()


def main():
    """Main function to run examples."""
    try:
        # Check if Redis is available
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379)
            r.ping()
            logger.info("✅ Redis connection successful")
        except Exception as e:
            logger.warning(f"⚠️  Redis not available: {e}")
            logger.warning("Running direct QuantGPT examples only...")
            example_queries()

        # Run both examples
        example_queries()
        example_service_client()

        logger.info("\n🎉 All examples completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main()
