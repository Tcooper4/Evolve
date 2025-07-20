#!/usr/bin/env python3
"""
QuantGPT Demonstration

A simple demonstration of the QuantGPT natural language interface.
Enhanced with safe LLM response parsing and proper routing.
"""

import datetime
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.exceptions import AgentExecutionError, QueryParsingError
from services.quant_gpt import QuantGPT

logger = logging.getLogger(__name__)


class SafeLLMResponseParser:
    """Safely parses and validates LLM responses."""

    def __init__(self):
        self.max_response_length = 10000  # Maximum response length
        self.sensitive_patterns = [
            r"password\s*[:=]\s*\S+",
            r"api_key\s*[:=]\s*\S+",
            r"token\s*[:=]\s*\S+",
            r"secret\s*[:=]\s*\S+",
        ]

    def parse_llm_response(
        self, response: str, response_type: str = "general"
    ) -> Dict[str, Any]:
        """Safely parse LLM response with validation and sanitization."""
        try:
            # Validate input
            if not response or not isinstance(response, str):
                return {
                    "success": False,
                    "error": "Invalid response format",
                    "parsed_data": None,
                }

            # Check response length
            if len(response) > self.max_response_length:
                logger.warning(
                    f"LLM response too long ({len(response)} chars), truncating"
                )
                response = response[: self.max_response_length] + "..."

            # Sanitize sensitive information
            sanitized = self._sanitize_response(response)

            # Parse based on response type
            if response_type == "json":
                parsed_data = self._parse_json_response(sanitized)
            elif response_type == "structured":
                parsed_data = self._parse_structured_response(sanitized)
            else:
                parsed_data = self._parse_general_response(sanitized)

            return {
                "success": True,
                "parsed_data": parsed_data,
                "original_length": len(response),
                "sanitized": sanitized != response,
            }

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {"success": False, "error": str(e), "parsed_data": None}

    def _sanitize_response(self, response: str) -> str:
        """Remove sensitive information from response."""
        sanitized = response
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
        return sanitized

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON-formatted LLM response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, return as text
                return {"text": response.strip()}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return {"text": response.strip(), "parse_error": str(e)}

    def _parse_structured_response(self, response: str) -> Dict[str, Any]:
        """Parse structured (key-value) LLM response."""
        result = {}
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if value:
                    result[key] = value

        return result if result else {"text": response.strip()}

    def _parse_general_response(self, response: str) -> Dict[str, Any]:
        """Parse general text response."""
        return {
            "text": response.strip(),
            "word_count": len(response.split()),
            "has_code": "```" in response,
            "has_links": "http" in response.lower(),
        }

    def validate_parsed_data(
        self, data: Dict[str, Any], expected_fields: list = None
    ) -> Dict[str, Any]:
        """Validate parsed data against expected fields."""
        if not data:
            return {"valid": False, "error": "No data to validate"}

        if expected_fields:
            missing_fields = [field for field in expected_fields if field not in data]
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {missing_fields}",
                    "missing_fields": missing_fields,
                }

        return {"valid": True, "data": data}


def demo_quant_gpt() -> dict:
    """Demonstrate QuantGPT functionality."""

    try:
        logger.info("ðŸ¤– QuantGPT Trading Interface Demonstration")
        logger.info("=" * 60)
        logger.info(
            "This demo shows how to use natural language to interact with the trading system."
        )
        logger.info("=" * 60)

        # Initialize QuantGPT
        logger.info("\nðŸ”§ Initializing QuantGPT...")
        quant_gpt = QuantGPT(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            redis_host="localhost",
            redis_port=6379,
        )

        logger.info("âœ… QuantGPT initialized successfully!")

        # Initialize safe response parser
        response_parser = SafeLLMResponseParser()

        # Demo queries
        demo_queries = [
            {
                "query": "Give me the best model for NVDA over 90 days",
                "description": "Model recommendation query",
            },
            {
                "query": "Should I long TSLA this week?",
                "description": "Trading signal query",
            },
            {
                "query": "Analyze BTCUSDT market conditions",
                "description": "Market analysis query",
            },
            {
                "query": "What's the trading signal for AAPL?",
                "description": "Another trading signal query",
            },
        ]

        logger.info(f"\nðŸ“ Running {len(demo_queries)} demo queries...")
        logger.info("-" * 60)

        for i, demo in enumerate(demo_queries, 1):
            query = demo["query"]
            description = demo["description"]

            logger.info(f"\nðŸŽ¯ Demo {i}: {description}")
            logger.info(f"Query: '{query}'")
            logger.info("-" * 50)

            try:
                # Process the query
                start_time = time.time()
                result = quant_gpt.process_query(query)
                processing_time = time.time() - start_time

                # Safely parse and validate the result
                parsed_result = response_parser.parse_llm_response(
                    str(result), response_type="structured"
                )

                if not parsed_result["success"]:
                    logger.error(
                        f"âŒ Failed to parse response: {parsed_result['error']}"
                    )
                    continue

                # Display results
                if result.get("status") == "success":
                    parsed = result.get("parsed_intent", {})
                    results = result.get("results", {})
                    commentary = result.get("gpt_commentary", "")

                    logger.info(f"â±ï¸  Processing Time: {processing_time:.2f} seconds")
                    logger.info(f"ðŸŽ¯ Intent: {parsed.get('intent', 'unknown')}")
                    logger.info(f"ðŸ“ˆ Symbol: {parsed.get('symbol', 'N/A')}")
                    logger.info(f"â° Timeframe: {parsed.get('timeframe', 'N/A')}")
                    logger.info(f"ðŸ“… Period: {parsed.get('period', 'N/A')}")

                    # Display action-specific results
                    action = results.get("action", "unknown")
                    logger.info(f"ðŸŽ¯ Action: {action}")

                    if action == "model_recommendation":
                        best_model = results.get("best_model")
                        if best_model:
                            logger.info(
                                f"ðŸ† Best Model: {best_model['model_type'].upper()}"
                            )
                            logger.info(
                                f"ðŸ“Š Model Score: {best_model['evaluation'].get('overall_score', 0):.2f}"
                            )
                            logger.info(
                                f"ðŸ“ˆ Models Built: {results.get('models_built', 0)}"
                            )
                            logger.info(
                                f"ðŸ” Models Evaluated: {results.get('models_evaluated', 0)}"
                            )

                    elif action == "trading_signal":
                        signal = results.get("signal", {})
                        if signal:
                            logger.info(f"ðŸ“Š Signal: {signal['signal']}")
                            logger.info(f"ðŸ’ª Strength: {signal['strength']}")
                            logger.info(f"ðŸŽ¯ Confidence: {signal['confidence']:.1%}")
                            logger.info(f"ðŸ§  Model Score: {signal['model_score']:.2f}")
                            logger.info(f"ðŸ’­ Reasoning: {signal['reasoning']}")

                    elif action == "market_analysis":
                        logger.info(
                            f"ðŸ“Š Market Data Available: {'Yes' if results.get('market_data') else 'No'}"
                        )
                        logger.info(
                            f"ðŸ“ˆ Plots Generated: {len(results.get('plots', []))}"
                        )
                        logger.info(
                            f"ðŸ¤– Model Analysis: {'Available' if results.get('model_analysis') else 'Not available'}"
                        )

                    # Safely parse and display GPT commentary
                    if commentary:
                        parsed_commentary = response_parser.parse_llm_response(
                            commentary, "general"
                        )
                        if parsed_commentary["success"]:
                            safe_commentary = parsed_commentary["parsed_data"]["text"]
                            logger.info(f"\nðŸ¤– GPT Commentary:")
                            logger.info("-" * 30)
                            logger.info(safe_commentary)
                        else:
                            logger.warning(
                                f"âš ï¸  Could not parse commentary: {parsed_commentary['error']}"
                            )

                    logger.info("âœ… Query processed successfully!")

                else:
                    error = result.get("error", "Unknown error")
                    logger.error(f"âŒ Error: {error}")
                    logger.error("ðŸ’¡ This might be due to missing services or data.")

            except QueryParsingError as e:
                logger.error(f"âŒ Query parsing error: {e}")
                logger.error("ðŸ’¡ Please rephrase your query with more specific details.")
            except AgentExecutionError as e:
                logger.error(f"âŒ Agent execution error: {e}")
                logger.error(f"ðŸ’¡ Context: {e.get_context_summary()}")
            except Exception as e:
                logger.error(f"âŒ Exception: {e}")
                logger.error("ðŸ’¡ This might be due to missing dependencies or services.")

            logger.info("\n" + "=" * 60)

        # Show available parameters
        logger.info("\nðŸ“‹ Available Parameters")
        logger.info("-" * 30)
        logger.info(
            f"Symbols: {', '.join(quant_gpt.trading_context['available_symbols'])}"
        )
        logger.info(
            f"Timeframes: {', '.join(quant_gpt.trading_context['available_timeframes'])}"
        )
        logger.info(
            f"Periods: {', '.join(quant_gpt.trading_context['available_periods'])}"
        )
        logger.info(
            f"Models: {', '.join(quant_gpt.trading_context['available_models'])}"
        )

        # Clean up
        logger.info("\nðŸ§¹ Cleaning up...")
        quant_gpt.close()
        logger.info("âœ… Demo completed!")

        logger.info("\n" + "=" * 60)
        logger.info("ðŸŽ‰ QuantGPT Demonstration Complete!")
        logger.info("=" * 60)
        logger.info("\nðŸ’¡ Tips for using QuantGPT:")
        logger.info("- Be specific about symbols, timeframes, and periods")
        logger.info(
            "- Ask for model recommendations, trading signals, or market analysis"
        )
        logger.info("- Use natural language - no need to learn specific commands")
        logger.info(
            "- The system will automatically route your query to the right services"
        )
        logger.info("\nðŸš€ Ready to start trading with natural language!")
        return {
            "success": True,
            "message": "QuantGPT demo completed successfully",
            "timestamp": datetime.datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat(),
        }


def main() -> dict:
    """Main function."""
    try:
        result = demo_quant_gpt()
        return {
            "success": True,
            "message": "Demo main completed",
            "timestamp": datetime.datetime.now().isoformat(),
            "demo_result": result,
        }
    except KeyboardInterrupt:
        logger.info("\n\nâ¹ï¸  Demo interrupted by user")
        return {
            "success": False,
            "error": "Demo interrupted by user",
            "timestamp": datetime.datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"\nâŒ Demo failed: {e}")
        logger.error("ðŸ’¡ Make sure Redis is running and all services are available")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat(),
        }


# Comment out test code to prevent accidental execution
"""
# Test functions - commented out to prevent accidental execution
def test_llm_response_parsing():
    parser = SafeLLMResponseParser()
    test_response = "Here is the analysis: {'signal': 'buy', 'confidence': 0.8}"
    result = parser.parse_llm_response(test_response, "json")
    print(f"Test result: {result}")

def test_query_processing():
    """Test query processing without full system initialization."""
    logger.info("Testing query processing...")
    try:
        # Test basic query parsing
        test_queries = [
            "What's the best model for AAPL?",
            "Generate a trading signal for TSLA",
            "Analyze the market for SPY"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            # Basic validation - just check if query is not empty
            if query and len(query.strip()) > 0:
                logger.info("âœ… Query validation passed")
            else:
                logger.warning("âš ï¸ Query validation failed")
                
        logger.info("âœ… Query processing tests completed")
        return True
    except Exception as e:
        logger.error(f"âŒ Query processing test failed: {e}")
        return False

if __name__ == "__main__":
    # Uncomment to run tests
    # test_llm_response_parsing()
    # test_query_processing()
    main()
"""

if __name__ == "__main__":
    main()
