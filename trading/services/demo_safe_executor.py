#!/usr/bin/env python3
"""
Safe Executor Demonstration

Demonstrates safe execution of user-defined models and strategies.
Enhanced with proper main() method structure and agent orchestration validation.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.service_client import ServiceClient

logger = logging.getLogger(__name__)


@dataclass
class AgentOrchestrationResult:
    """Result from agent orchestration execution."""

    success: bool
    output_type: str
    data: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DemoSafeExecutor:
    """Demonstration class for safe executor functionality."""

    def __init__(self):
        """Initialize the demo executor."""
        self.client = None
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("demo_safe_executor.log"),
            ],
        )

    def initialize_client(self) -> bool:
        """Initialize ServiceClient."""
        try:
            logger.info("🔧 Initializing ServiceClient...")
            self.client = ServiceClient(redis_host="localhost", redis_port=6379)
            logger.info("✅ ServiceClient initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize ServiceClient: {e}")
            return False

    def validate_agent_orchestration_output(
        self, result: Dict[str, Any]
    ) -> AgentOrchestrationResult:
        """Validate agent orchestration output type and structure.

        Args:
            result: Result from agent orchestration

        Returns:
            Validated orchestration result
        """
        try:
            # Validate basic structure
            if not isinstance(result, dict):
                return AgentOrchestrationResult(
                    success=False,
                    output_type="invalid",
                    data={},
                    execution_time=0.0,
                    error_message="Result is not a dictionary",
                )

            # Extract and validate output type
            output_type = result.get("type", "unknown")
            valid_types = [
                "model_executed",
                "strategy_executed",
                "indicator_executed",
                "agent_orchestrated",
            ]

            if output_type not in valid_types:
                return AgentOrchestrationResult(
                    success=False,
                    output_type=output_type,
                    data=result,
                    execution_time=0.0,
                    error_message=f"Invalid output type: {output_type}. Expected one of {valid_types}",
                )

            # Validate execution result structure
            execution_result = result.get("result", {})
            if not isinstance(execution_result, dict):
                return AgentOrchestrationResult(
                    success=False,
                    output_type=output_type,
                    data=result,
                    execution_time=0.0,
                    error_message="Execution result is not a dictionary",
                )

            # Extract execution details
            status = execution_result.get("status", "unknown")
            execution_time = execution_result.get("execution_time", 0.0)
            return_value = execution_result.get("return_value", {})
            error = execution_result.get("error")

            # Validate return value based on output type
            if status == "success":
                if not self._validate_return_value(output_type, return_value):
                    return AgentOrchestrationResult(
                        success=False,
                        output_type=output_type,
                        data=result,
                        execution_time=execution_time,
                        error_message="Invalid return value structure for output type",
                    )

            return AgentOrchestrationResult(
                success=status == "success",
                output_type=output_type,
                data=result,
                execution_time=execution_time,
                error_message=error,
                metadata={"status": status, "return_value": return_value},
            )

        except Exception as e:
            return AgentOrchestrationResult(
                success=False,
                output_type="error",
                data=result,
                execution_time=0.0,
                error_message=f"Validation error: {str(e)}",
            )

    def _validate_return_value(
        self, output_type: str, return_value: Dict[str, Any]
    ) -> bool:
        """Validate return value structure based on output type."""
        if not isinstance(return_value, dict):
            return False

        if output_type == "model_executed":
            # Models should have prediction-related fields
            required_fields = ["prediction", "confidence"]
            return all(field in return_value for field in required_fields)

        elif output_type == "strategy_executed":
            # Strategies should have signal-related fields
            required_fields = ["signal", "confidence"]
            return all(field in return_value for field in required_fields)

        elif output_type == "indicator_executed":
            # Indicators should have indicator-specific fields
            return len(return_value) > 0  # At least one indicator value

        elif output_type == "agent_orchestrated":
            # Agent orchestration should have decision-related fields
            return "decision" in return_value or "action" in return_value

        return True

    def demo_safe_model_execution(self) -> bool:
        """Demo 1: Safe Model Execution."""
        logger.info("\n🎯 Demo 1: Safe Model Execution")
        logger.info("-" * 40)

        model_code = """
import numpy as np
import pandas as pd

def main(input_data):
    # Simple moving average model
    prices = input_data.get('prices', [100, 101, 102, 103, 104])
    window = input_data.get('window', 3)

    if len(prices) < window:
        return {"error": "Not enough data"}

    ma = np.mean(prices[-window:])
    prediction = ma * 1.01  # Simple prediction

    return {
        "prediction": prediction,
        "moving_average": ma,
        "confidence": 0.7
    }
"""

        input_data = {"prices": [100, 101, 102, 103, 104, 105, 106], "window": 3}

        logger.info("Executing simple moving average model...")
        result = self.client.execute_model_safely(
            model_code=model_code,
            model_name="simple_ma_model",
            input_data=input_data,
            model_type="custom",
        )

        # Validate the result
        validated_result = self.validate_agent_orchestration_output(result)

        if validated_result.success:
            logger.info(f"✅ Model execution successful")
            logger.info(f"Output Type: {validated_result.output_type}")
            logger.info(f"Execution Time: {validated_result.execution_time:.2f}s")

            return_value = validated_result.metadata.get("return_value", {})
            logger.info(f"Prediction: {return_value.get('prediction', 'N/A')}")
            logger.info(f"Moving Average: {return_value.get('moving_average', 'N/A')}")
            logger.info(f"Confidence: {return_value.get('confidence', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Model execution failed")
            logger.error(f"Error: {validated_result.error_message}")
            return False

    def demo_safe_strategy_execution(self) -> bool:
        """Demo 2: Safe Strategy Execution."""
        logger.info("\n🎯 Demo 2: Safe Strategy Execution")
        logger.info("-" * 40)

        strategy_code = """
import numpy as np

def main(input_data):
    market_data = input_data.get('market_data', {})
    parameters = input_data.get('parameters', {})

    # Simple RSI strategy
    prices = market_data.get('prices', [100, 101, 102, 103, 104])
    rsi = market_data.get('rsi', 65)

    # Strategy logic
    if rsi > 70:
        signal = "SELL"
        confidence = 0.8
    elif rsi < 30:
        signal = "BUY"
        confidence = 0.8
    else:
        signal = "HOLD"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "rsi": rsi,
        "reasoning": f"RSI is {rsi}, indicating {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'} conditions"
    }
"""

        market_data = {"prices": [100, 101, 102, 103, 104, 105, 106], "rsi": 75}

        parameters = {"rsi_oversold": 30, "rsi_overbought": 70}

        logger.info("Executing RSI strategy...")
        result = self.client.execute_strategy_safely(
            strategy_code=strategy_code,
            strategy_name="rsi_strategy",
            market_data=market_data,
            parameters=parameters,
        )

        # Validate the result
        validated_result = self.validate_agent_orchestration_output(result)

        if validated_result.success:
            logger.info(f"✅ Strategy execution successful")
            logger.info(f"Output Type: {validated_result.output_type}")
            logger.info(f"Execution Time: {validated_result.execution_time:.2f}s")

            return_value = validated_result.metadata.get("return_value", {})
            logger.info(f"Signal: {return_value.get('signal', 'N/A')}")
            logger.info(f"Confidence: {return_value.get('confidence', 'N/A')}")
            logger.info(f"RSI: {return_value.get('rsi', 'N/A')}")
            logger.info(f"Reasoning: {return_value.get('reasoning', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Strategy execution failed")
            logger.error(f"Error: {validated_result.error_message}")
            return False

    def demo_safe_indicator_execution(self) -> bool:
        """Demo 3: Safe Indicator Execution."""
        logger.info("\n🎯 Demo 3: Safe Indicator Execution")
        logger.info("-" * 40)

        indicator_code = """
import numpy as np

def main(input_data):
    price_data = input_data.get('price_data', {})
    parameters = input_data.get('parameters', {})

    # Calculate MACD indicator
    prices = price_data.get('prices', [100, 101, 102, 103, 104, 105, 106])
    fast_period = parameters.get('fast_period', 12)
    slow_period = parameters.get('slow_period', 26)

    if len(prices) < slow_period:
        return {"error": "Not enough data for MACD calculation"}

    # Simple MACD calculation
    fast_ma = np.mean(prices[-fast_period:])
    slow_ma = np.mean(prices[-slow_period:])
    macd = fast_ma - slow_ma

    return {
        "macd": macd,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "signal": "BULLISH" if macd > 0 else "BEARISH"
    }
"""

        price_data = {
            "prices": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
                130,
            ]
        }

        parameters = {"fast_period": 12, "slow_period": 26}

        logger.info("Executing MACD indicator...")
        result = self.client.execute_indicator_safely(
            indicator_code=indicator_code,
            indicator_name="macd_indicator",
            price_data=price_data,
            parameters=parameters,
        )

        # Validate the result
        validated_result = self.validate_agent_orchestration_output(result)

        if validated_result.success:
            logger.info(f"✅ Indicator execution successful")
            logger.info(f"Output Type: {validated_result.output_type}")
            logger.info(f"Execution Time: {validated_result.execution_time:.2f}s")

            return_value = validated_result.metadata.get("return_value", {})
            logger.info(f"MACD: {return_value.get('macd', 'N/A')}")
            logger.info(f"Fast MA: {return_value.get('fast_ma', 'N/A')}")
            logger.info(f"Slow MA: {return_value.get('slow_ma', 'N/A')}")
            logger.info(f"Signal: {return_value.get('signal', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Indicator execution failed")
            logger.error(f"Error: {validated_result.error_message}")
            return False

    def demo_error_handling(self) -> bool:
        """Demo 4: Error Handling (Dangerous Code)."""
        logger.info("\n🎯 Demo 4: Error Handling (Dangerous Code)")
        logger.info("-" * 40)

        dangerous_code = """
import os
import subprocess

def main(input_data):
    # This should be blocked by the safe executor
    os.system("rm -rf /")  # Dangerous command
    return {"status": "dangerous"}
"""

        logger.info("Attempting to execute dangerous code...")
        result = self.client.execute_model_safely(
            model_code=dangerous_code,
            model_name="dangerous_model",
            input_data={},
            model_type="custom",
        )

        # Validate the result
        validated_result = self.validate_agent_orchestration_output(result)

        if not validated_result.success:
            logger.info(f"✅ Dangerous code properly blocked")
            logger.info(f"Error: {validated_result.error_message}")
            return True
        else:
            logger.error(f"❌ Dangerous code was not blocked!")
            return False

    def demo_agent_orchestration(self) -> bool:
        """Demo 5: Agent Orchestration."""
        logger.info("\n🎯 Demo 5: Agent Orchestration")
        logger.info("-" * 40)

        orchestration_code = """
def main(input_data):
    # Simulate agent orchestration decision
    market_condition = input_data.get('market_condition', 'neutral')
    risk_level = input_data.get('risk_level', 'medium')

    # Decision logic
    if market_condition == 'bullish' and risk_level == 'low':
        decision = "AGGRESSIVE_LONG"
        confidence = 0.9
    elif market_condition == 'bearish' and risk_level == 'high':
        decision = "DEFENSIVE_SHORT"
        confidence = 0.8
    else:
        decision = "NEUTRAL"
        confidence = 0.6

    return {
        "decision": decision,
        "confidence": confidence,
        "market_condition": market_condition,
        "risk_level": risk_level,
        "reasoning": f"Market is {market_condition} with {risk_level} risk"
    }
"""

        input_data = {"market_condition": "bullish", "risk_level": "low"}

        logger.info("Executing agent orchestration...")
        result = self.client.execute_model_safely(
            model_code=orchestration_code,
            model_name="agent_orchestration",
            input_data=input_data,
            model_type="orchestration",
        )

        # Validate the result
        validated_result = self.validate_agent_orchestration_output(result)

        if validated_result.success:
            logger.info(f"✅ Agent orchestration successful")
            logger.info(f"Output Type: {validated_result.output_type}")
            logger.info(f"Execution Time: {validated_result.execution_time:.2f}s")

            return_value = validated_result.metadata.get("return_value", {})
            logger.info(f"Decision: {return_value.get('decision', 'N/A')}")
            logger.info(f"Confidence: {return_value.get('confidence', 'N/A')}")
            logger.info(
                f"Market Condition: {return_value.get('market_condition', 'N/A')}"
            )
            logger.info(f"Risk Level: {return_value.get('risk_level', 'N/A')}")
            logger.info(f"Reasoning: {return_value.get('reasoning', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Agent orchestration failed")
            logger.error(f"Error: {validated_result.error_message}")
            return False

    def run_all_demos(self) -> Dict[str, bool]:
        """Run all demonstration scenarios."""
        logger.info("🛡️ Safe Executor Demonstration")
        logger.info("=" * 60)
        logger.info(
            "This demo shows how to safely execute user-defined models and strategies."
        )
        logger.info("=" * 60)

        # Initialize client
        if not self.initialize_client():
            return {"initialization": False}

        # Run all demos
        results = {
            "model_execution": self.demo_safe_model_execution(),
            "strategy_execution": self.demo_safe_strategy_execution(),
            "indicator_execution": self.demo_safe_indicator_execution(),
            "error_handling": self.demo_error_handling(),
            "agent_orchestration": self.demo_agent_orchestration(),
        }

        # Summary
        logger.info("\n📊 Demo Summary")
        logger.info("=" * 40)
        for demo_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{demo_name.replace('_', ' ').title()}: {status}")

        passed = sum(results.values())
        total = len(results)
        logger.info(f"\nOverall: {passed}/{total} demos passed")

        return results


def main():
    """Main entry point for demo execution."""
    try:
        logger.info("🚀 Starting Safe Executor Demo")
        logger.info("=" * 50)

        # Initialize demo executor
        demo = DemoSafeExecutor()

        # Initialize client
        if not demo.initialize_client():
            logger.error("Failed to initialize client. Exiting.")
            return 1

        # Run all demos
        results = demo.run_all_demos()

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 Demo Summary")
        logger.info("=" * 50)

        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)

        for demo_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} {demo_name}")

        logger.info(f"\nOverall: {success_count}/{total_count} demos passed")

        if success_count == total_count:
            logger.info("🎉 All demos completed successfully!")
            return 0
        else:
            logger.warning("⚠️  Some demos failed. Check logs for details.")
            return 1

    except KeyboardInterrupt:
        logger.info("\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Unexpected error in main: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
