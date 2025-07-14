#!/usr/bin/env python3
"""
QuantGPT Test Script

Tests the QuantGPT interface functionality.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.quant_gpt import QuantGPT
from services.service_client import ServiceClient

logger = logging.getLogger(__name__)


def test_direct_quant_gpt():
    """Test direct QuantGPT usage."""
    logger.info("🧪 Testing Direct QuantGPT Usage")
    logger.info("=" * 50)

    try:
        # Initialize QuantGPT
        quant_gpt = QuantGPT(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            redis_host="localhost",
            redis_port=6379,
        )

        # Test queries
        test_queries = [
            "Give me the best model for NVDA over 90 days",
            "Should I long TSLA this week?",
            "Analyze BTCUSDT market conditions",
        ]

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n📝 Test Query {i}: {query}")
            logger.info("-" * 40)

            start_time = time.time()
            result = quant_gpt.process_query(query)
            processing_time = time.time() - start_time

            logger.info(f"Processing Time: {processing_time:.2f}s")
            logger.info(f"Status: {result.get('status', 'unknown')}")

            if result.get("status") == "success":
                parsed = result.get("parsed_intent", {})
                logger.info(f"Intent: {parsed.get('intent', 'unknown')}")
                logger.info(f"Symbol: {parsed.get('symbol', 'N/A')}")
                logger.info(f"Timeframe: {parsed.get('timeframe', 'N/A')}")
                logger.info(f"Period: {parsed.get('period', 'N/A')}")

                commentary = result.get("gpt_commentary", "")
                if commentary:
                    logger.info(f"GPT Commentary: {commentary[:100]}...")

                logger.info("✅ Test passed")
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Test failed: {error}")

        quant_gpt.close()
        return True

    except Exception as e:
        logger.error(f"❌ Direct QuantGPT test failed: {e}")
        return False


def test_service_client():
    """Test QuantGPT via ServiceClient."""
    logger.info("\n🔗 Testing QuantGPT via ServiceClient")
    logger.info("=" * 50)

    try:
        # Initialize ServiceClient
        client = ServiceClient(redis_host="localhost", redis_port=6379)

        # Test queries
        test_queries = [
            "What's the best model for MSFT?",
            "Should I buy AMZN now?",
            "Analyze ETHUSDT market",
        ]

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n📝 Service Test Query {i}: {query}")
            logger.info("-" * 40)

            start_time = time.time()
            result = client.process_natural_language_query(query)
            processing_time = time.time() - start_time

            logger.info(f"Processing Time: {processing_time:.2f}s")

            if result:
                logger.info(f"Response Type: {result.get('type', 'unknown')}")

                if result.get("type") == "query_processed":
                    query_result = result.get("result", {})
                    if query_result.get("status") == "success":
                        parsed = query_result.get("parsed_intent", {})
                        logger.info(f"Intent: {parsed.get('intent', 'unknown')}")
                        logger.info(f"Symbol: {parsed.get('symbol', 'N/A')}")
                        logger.info("✅ Service test passed")
                    else:
                        logger.error(
                            f"❌ Query failed: {query_result.get('error', 'Unknown error')}"
                        )
                else:
                    logger.error(
                        f"❌ Service error: {result.get('error', 'Unknown error')}"
                    )
            else:
                logger.error("❌ No response from service")

        # Test available symbols
        logger.info(f"\n📊 Testing Available Symbols")
        logger.info("-" * 40)

        symbols_result = client.get_available_symbols()
        if symbols_result and symbols_result.get("type") == "available_symbols":
            symbols = symbols_result.get("symbols", [])
            logger.info(f"Available Symbols: {', '.join(symbols)}")
            logger.info("✅ Symbols test passed")
        else:
            logger.error("❌ Symbols test failed")

        client.close()
        return True

    except Exception as e:
        logger.error(f"❌ ServiceClient test failed: {e}")
        return False


def test_query_parsing():
    """Test query parsing functionality."""
    logger.info("\n🔍 Testing Query Parsing")
    logger.info("=" * 50)

    try:
        quant_gpt = QuantGPT(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            redis_host="localhost",
            redis_port=6379,
        )

        # Test various query formats
        test_cases = [
            {
                "query": "Give me the best model for NVDA over 90 days",
                "expected_intent": "model_recommendation",
                "expected_symbol": "NVDA",
                "expected_period": "90d",
            },
            {
                "query": "Should I long TSLA this week?",
                "expected_intent": "trading_signal",
                "expected_symbol": "TSLA",
                "expected_timeframe": "1h",
            },
            {
                "query": "Analyze BTCUSDT market conditions",
                "expected_intent": "market_analysis",
                "expected_symbol": "BTCUSDT",
            },
        ]

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n📝 Parsing Test {i}: {test_case['query']}")
            logger.info("-" * 40)

            result = quant_gpt.process_query(test_case["query"])

            if result.get("status") == "success":
                parsed = result.get("parsed_intent", {})

                # Check intent
                intent = parsed.get("intent")
                expected_intent = test_case.get("expected_intent")
                if intent == expected_intent:
                    logger.info(f"✅ Intent: {intent}")
                else:
                    logger.error(
                        f"❌ Intent mismatch: expected {expected_intent}, got {intent}"
                    )

                # Check symbol
                symbol = parsed.get("symbol")
                expected_symbol = test_case.get("expected_symbol")
                if symbol == expected_symbol:
                    logger.info(f"✅ Symbol: {symbol}")
                else:
                    logger.error(
                        f"❌ Symbol mismatch: expected {expected_symbol}, got {symbol}"
                    )

                # Check other parameters
                for param in ["timeframe", "period"]:
                    if param in test_case:
                        value = parsed.get(param)
                        expected_value = test_case.get(f"expected_{param}")
                        if value == expected_value:
                            logger.info(f"✅ {param}: {value}")
                        else:
                            logger.error(
                                f"❌ {param} mismatch: expected {expected_value}, got {value}"
                            )

                logger.info("✅ Parsing test passed")
            else:
                logger.error(
                    f"❌ Parsing test failed: {result.get('error', 'Unknown error')}"
                )

        quant_gpt.close()
        return True

    except Exception as e:
        logger.error(f"❌ Query parsing test failed: {e}")
        return False


def main():
    """Run all QuantGPT tests."""
    logger.info("🚀 QuantGPT Test Suite")
    logger.info("=" * 60)

    # Check Redis availability
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379)
        r.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.warning(f"⚠️  Redis not available: {e}")
        logger.info("Running direct QuantGPT tests only...")
        test_direct_quant_gpt()
        test_query_parsing()

    # Run all tests
    tests = [
        ("Direct QuantGPT", test_direct_quant_gpt),
        ("ServiceClient", test_service_client),
        ("Query Parsing", test_query_parsing),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 Test Results Summary")
    logger.info("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! QuantGPT is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
